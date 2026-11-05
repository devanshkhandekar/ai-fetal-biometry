import Keycloak from 'keycloak-js'

export const keycloak = new Keycloak({
  url: import.meta.env.VITE_KEYCLOAK_URL || 'http://localhost:8081',
  realm: import.meta.env.VITE_KEYCLOAK_REALM || 'fetal-local',
  clientId: import.meta.env.VITE_KEYCLOAK_CLIENT_ID || 'fetal-web',
})

export async function initAuth() {
  return keycloak.init({ onLoad: 'login-required', pkceMethod: 'S256', checkLoginIframe: false })
}
